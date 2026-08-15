"""End-to-end smoke test against a running docker-compose stack.

Exercises the real thing the in-process tests cannot: presigned upload to
object storage, the queue between containers, the WebSocket, whether the media
is actually gone from the bucket afterwards, and whether the audit trail is
really in Postgres -- read back from the columns, not taken from the API
response that the same process just produced.

That last part is not belt-and-braces. Every in-process test runs against
FakeDb, which cannot reproduce psycopg3's Connection/Cursor split, so a DB
write path can be broken while the whole suite stays green. It already
happened once: insert_items called executemany on a Connection and
dead-lettered every job.

    docker compose up --build -d
    python scripts/smoke_compose.py

Exits non-zero on the first failure so it can gate a "compose runs end to end"
claim rather than being read by eye.
"""
from __future__ import annotations

import json
import os
import pathlib
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

GATEWAY = os.environ.get("DF_GATEWAY", "http://localhost:8000")
TIMEOUT_SECONDS = 120

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def _req(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{GATEWAY}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode()}


def _psql(sql: str) -> list[list[str]]:
    """Run a query against the real Postgres and return raw rows.

    Reached via `compose exec` rather than a published port: postgres sits on
    the `internal` network deliberately (docker-compose.yml), and punching a
    host port through just to test would weaken the thing being tested.

    This exists because the API response is produced by the same process that
    wrote the row, so checking it proves only that the process is
    self-consistent. Reading the columns back is what catches a DB-layer
    regression -- insert_items calling executemany on a psycopg3 Connection got
    all the way to a live stack precisely because every in-process test runs
    against FakeDb.
    """
    proc = subprocess.run(
        ["docker", "compose", "exec", "-T", "postgres", "psql",
         "-U", "deepfake", "-d", "deepfake", "-t", "-A", "-F", "\x1f", "-c", sql],
        capture_output=True, text=True, timeout=60,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return [ln.split("\x1f") for ln in proc.stdout.strip().splitlines() if ln]


def _multipart(fields: dict, payload: bytes, content_type: str) -> tuple[bytes, str]:
    """Encode an S3 POST-policy upload.

    The file part must come last -- S3 ignores any form field that follows it,
    which would silently drop the policy and signature.
    """
    boundary = "----dfsmoke" + secrets.token_hex(12)
    out = bytearray()
    for name, value in fields.items():
        out += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    out += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="upload.bin"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    out += payload
    out += f"\r\n--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


def _post_grant(grant, payload: bytes, content_type: str) -> int:
    """POST a payload against a presigned grant, returning the HTTP status."""
    body, ct = _multipart(grant.fields, payload, content_type)
    req = urllib.request.Request(
        grant.url, data=body, method="POST", headers={"Content-Type": ct}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def probe_size_cap(storage) -> None:
    """Prove object storage actually enforces content-length-range.

    Deliberately out-of-band: the grant is minted by calling presign_upload
    directly with a small bound, never through the gateway. The gateway's cap
    is a fixed Tier 1 value that no caller gets to vary -- adding a
    "smaller cap for testing" parameter would punch a hole in precisely the
    limit this check defends, and test-only exceptions to a security boundary
    have a poor record of staying test-only.

    The property under test is "does storage honour the condition", not "does
    it honour it at exactly DF_MAX_UPLOAD_BYTES", so a small bound tests it
    exactly as well and runs on every smoke run rather than only when someone
    remembers to lower .env.

    Both halves are required. Without the accepted case, a rejection would
    only prove the grant was broken somehow -- not that the bound was what
    stopped it.
    """
    bound = 1024
    ctype = "application/octet-stream"
    key = f"probe/size-cap-{secrets.token_hex(8)}"

    under = _post_grant(storage.presign_upload(key, ctype, bound), b"u" * (bound // 2), ctype)
    check("storage accepts a body within the signed bound", 200 <= under < 300, str(under))
    check("...and the object landed", storage.exists(key))
    storage.delete_object(key)

    over = _post_grant(storage.presign_upload(key, ctype, bound), b"o" * (bound + 1), ctype)
    check("storage rejects a body over the signed bound", over in (400, 403), str(over))
    check("rejected body left nothing in the bucket", not storage.exists(key))


def main() -> int:
    from df.config import settings
    from df.storage import S3Storage

    # One client for the out-of-band cap probe and the later bucket
    # assertions. Points at the public endpoint because this script runs on
    # the host, outside the compose network.
    storage = S3Storage(endpoint=settings.s3_public_endpoint)

    print(f"gateway: {GATEWAY}\n")

    status, health = _req("GET", "/healthz")
    check("gateway healthy", status == 200 and health.get("ok"), str(health))
    if health.get("inference_backend") == "stub":
        print("      note: stub backend -- scores are placeholders, not detections\n")

    # 1. presigned upload
    status, job = _req(
        "POST", "/v1/jobs", {"media_type": "video", "content_type": "video/mp4"}
    )
    check("job created", status == 201 and "job_id" in job, str(job))
    job_id, upload_url = job["job_id"], job["upload_url"]
    check("presigned URL returned", upload_url.startswith("http"), upload_url[:60])

    # 2. upload straight to object storage, bypassing the API
    payload = b"smoke-test-media-bytes" * 512
    body, content_type = _multipart(job["upload_fields"], payload, "video/mp4")
    post = urllib.request.Request(
        upload_url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(post, timeout=60) as resp:
            check("presigned upload accepted", 200 <= resp.status < 300, str(resp.status))
    except urllib.error.HTTPError as exc:
        check("presigned upload accepted", False, f"{exc.code} {exc.read()[:200]!r}")

    # 2b. the size cap is a signed condition, not a client-side promise. These
    # bytes never pass through the gateway, so if storage does not reject an
    # oversized body nothing else will. Probed out-of-band -- see probe_size_cap.
    probe_size_cap(storage)

    # 3. notify -> enqueue
    status, ack = _req("POST", f"/v1/jobs/{job_id}/uploaded")
    check("upload acknowledged and enqueued", status == 202 and ack.get("enqueued"), str(ack))

    status, again = _req("POST", f"/v1/jobs/{job_id}/uploaded")
    check("notify is idempotent", again.get("enqueued") is False, str(again))

    # 4. poll to completion (the reconnect fallback path)
    deadline = time.time() + TIMEOUT_SECONDS
    doc: dict = {}
    while time.time() < deadline:
        _, doc = _req("GET", f"/v1/jobs/{job_id}")
        if doc.get("status") in {"complete", "failed", "dead_letter"}:
            break
        time.sleep(2)

    check("job reached a terminal state", doc.get("status") == "complete", str(doc.get("status")))
    print(f"\n{json.dumps(doc, indent=2)}\n")

    # 5. the audit trail
    check("content hash recorded", bool(doc.get("content_hash")))
    check("model version recorded", bool(doc.get("model_version_id")))
    check("aggregation method recorded", bool(doc.get("aggregation_method")))
    check("aggregation params recorded", bool(doc.get("aggregation_params")))
    check("a verdict was reached", doc.get("result_class") is not None)

    # 5b. the audit trail as Postgres actually holds it, not as the API
    # describes it. job_id is a UUID minted by our own API, so interpolating it
    # is safe here.
    rows = _psql(
        "SELECT content_hash, model_version_id, aggregation_method, "
        "aggregation_params::text, status "
        f"FROM jobs WHERE id = '{job_id}';"
    )
    check("job row readable from postgres", len(rows) == 1, f"{len(rows)} rows")
    db_hash, db_model, db_method, db_params, db_status = rows[0]

    check("db status is complete", db_status == "complete", db_status)
    check("db content_hash agrees with the API", db_hash == doc.get("content_hash"), db_hash)
    check("db content_hash is a sha256", len(db_hash) == 64, f"len={len(db_hash)}")
    check("db model_version_id recorded", bool(db_model), repr(db_model))
    check("db aggregation_method recorded", bool(db_method), repr(db_method))
    try:
        parsed = json.loads(db_params)
    except (ValueError, TypeError):
        parsed = None
    check("db aggregation_params is usable json", isinstance(parsed, dict) and bool(parsed),
          repr(db_params)[:80])

    # The write path that shipped broken. A green pytest run cannot see this:
    # FakeDb has no Connection/Cursor split to get wrong.
    counts = _psql(
        "SELECT count(*), count(score), count(confidence) "
        f"FROM job_items WHERE job_id = '{job_id}';"
    )
    total, scored, confident = (int(v) for v in counts[0])
    # total == 0 means insert_items wrote nothing.
    check("per-item scores written to postgres", total > 0, f"{total} rows")
    check("every item row carries a score", scored == total, f"{scored}/{total}")
    check("every item row carries a confidence", confident == total, f"{confident}/{total}")
    print(f"      {total} job_items rows written\n")

    # 6. Tier 1
    check("media reported deleted", doc.get("media_deleted") is True, str(doc.get("media_deleted")))
    check(
        "adversarial advisory present",
        any("adversarial" in a.lower() for a in doc.get("advisories", [])),
    )

    # 7. media really is gone from the bucket, not just flagged in Postgres
    check(
        "raw upload absent from bucket",
        not storage.exists(f"raw/{job_id}/original"),
        "object still present",
    )
    check(
        "working face crops absent from bucket",
        storage.list_prefix(f"derived/{job_id}/") == [],
        "derived objects still present",
    )

    # 7b. extended retention window: a >80 result must have preserved the crops
    # that drove the score; anything else must have preserved nothing.
    cold = storage.list_prefix(f"cold/{job_id}/")
    if doc.get("band") == "likely_manipulated":
        check("high band preserved its driving crops", len(cold) > 0, "cold storage empty")
        check(
            "extended retention window recorded",
            doc.get("extended_retention_until") is not None,
        )
        check(
            "window is described as a fixed timer, not a hold",
            any("not a legal hold" in a for a in doc.get("advisories", [])),
        )
    else:
        check(
            f"non-high band ({doc.get('band')}) preserved nothing",
            cold == [],
            f"unexpected cold objects: {cold}",
        )

    # 8. rate limiter actually rejects
    codes = [_req("POST", "/v1/jobs", {"media_type": "image"})[0] for _ in range(60)]
    check("rate limiter returns 429 under burst", 429 in codes, f"codes seen: {sorted(set(codes))}")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
