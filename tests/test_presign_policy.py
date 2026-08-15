"""The presigned upload grant carries a real size cap.

The compose smoke test proves MinIO *enforces* the cap. Nothing there proves
the code still *asks* for it -- drop the condition, or sign it with the wrong
bound, and the smoke test's oversized probe would still be rejected for some
other reason, or quietly start passing.

So these tests decode the base64 policy document boto3 actually signed and
assert on its contents. Real signing, no mocks and no network:
generate_presigned_post is a local operation, so this runs in milliseconds and
still exercises the production code path.

Why it matters (CLAUDE.md, Tier 1): these bytes never traverse the gateway.
Ingress rate limiting never sees them, and the API cannot check a length it
never receives. The policy condition is the only enforcement point that exists.
"""
from __future__ import annotations

import base64
import json

import pytest

from df import storage as storage_mod
from df.config import Settings

INTERNAL = "http://minio:9000"
PUBLIC = "http://localhost:9000"
KEY = "raw/job-abc/original"


@pytest.fixture
def s3(monkeypatch):
    """A real S3Storage signing against fake credentials.

    Nothing here talks to a network; boto3 signs locally.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("DF_S3_ENDPOINT", INTERNAL)
    monkeypatch.setenv("DF_S3_PUBLIC_ENDPOINT", PUBLIC)
    monkeypatch.setenv("DF_S3_BUCKET", "test-bucket")
    # Settings reads os.environ at construction, and storage.py holds a
    # module-level reference to the singleton.
    monkeypatch.setattr(storage_mod, "settings", Settings())
    return storage_mod.S3Storage()


def policy_of(grant) -> dict:
    return json.loads(base64.b64decode(grant.fields["policy"]))


def length_range(grant) -> list | None:
    for cond in policy_of(grant)["conditions"]:
        if isinstance(cond, list) and cond and cond[0] == "content-length-range":
            return cond
    return None


def test_policy_carries_a_content_length_range(s3):
    grant = s3.presign_upload(KEY, "video/mp4", 4096)

    assert length_range(grant) is not None, (
        "no content-length-range in the signed policy -- the upload grant is "
        "an unbounded write to the bucket"
    )


def test_upper_bound_is_the_requested_max(s3):
    grant = s3.presign_upload(KEY, "video/mp4", 4096)

    assert length_range(grant)[2] == 4096


def test_upper_bound_tracks_the_argument(s3):
    """Catches a hardcoded constant that happens to match one test's value."""
    small = length_range(s3.presign_upload(KEY, "video/mp4", 1024))
    large = length_range(s3.presign_upload(KEY, "video/mp4", 2 * 1024**3))

    assert small[2] == 1024
    assert large[2] == 2 * 1024**3


def test_zero_byte_uploads_are_excluded(s3):
    """A 0-byte object would complete the job and score nothing."""
    assert length_range(s3.presign_upload(KEY, "video/mp4", 4096))[1] >= 1


def test_policy_pins_the_content_type(s3):
    grant = s3.presign_upload(KEY, "video/mp4", 4096)
    conditions = policy_of(grant)["conditions"]

    assert {"Content-Type": "video/mp4"} in conditions
    assert grant.fields["Content-Type"] == "video/mp4"


def test_policy_pins_the_object_key(s3):
    """Without this the grant is a write to anywhere in the bucket."""
    grant = s3.presign_upload(KEY, "video/mp4", 4096)

    assert {"key": KEY} in policy_of(grant)["conditions"]
    assert grant.fields["key"] == KEY


def test_grant_is_signed_against_the_browser_reachable_host(s3):
    """CLAUDE.md Tier 1: the host is part of the signature. Signing against the
    internal name yields a URL no browser can use."""
    grant = s3.presign_upload(KEY, "video/mp4", 4096)

    assert grant.url.startswith(PUBLIC)
    assert not grant.url.startswith(INTERNAL)


def test_grant_is_a_post_with_signature_fields(s3):
    grant = s3.presign_upload(KEY, "video/mp4", 4096)

    assert grant.method == "POST"
    assert grant.fields["x-amz-signature"]
    assert grant.fields["x-amz-algorithm"] == "AWS4-HMAC-SHA256"
