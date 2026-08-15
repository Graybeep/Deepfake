"""End-to-end smoke test against a running docker-compose stack.

Exercises the real thing the in-process tests cannot: presigned upload to
object storage, the queue between containers, the WebSocket, and whether the
media is actually gone from the bucket afterwards.

    docker compose up --build -d
    python scripts/smoke_compose.py

Exits non-zero on the first failure so it can gate a "compose runs end to end"
claim rather than being read by eye.
"""
from __future__ import annotations

import json
import os
import pathlib
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


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


def main() -> int:
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
    put = urllib.request.Request(
        upload_url, data=payload, method="PUT", headers={"Content-Type": "video/mp4"}
    )
    try:
        with urllib.request.urlopen(put, timeout=60) as resp:
            check("presigned upload accepted", 200 <= resp.status < 300, str(resp.status))
    except urllib.error.HTTPError as exc:
        check("presigned upload accepted", False, f"{exc.code} {exc.read()[:200]!r}")

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

    # 6. Tier 1
    check("media reported deleted", doc.get("media_deleted") is True, str(doc.get("media_deleted")))
    check(
        "adversarial advisory present",
        any("adversarial" in a.lower() for a in doc.get("advisories", [])),
    )

    # 7. media really is gone from the bucket, not just flagged in Postgres
    from df.config import settings
    from df.storage import S3Storage

    storage = S3Storage(endpoint=settings.s3_public_endpoint)
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
