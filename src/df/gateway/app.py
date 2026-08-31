"""Public API.

Ingest is presigned-upload only: the client POSTs bytes straight to S3 and the
API never proxies media (Tier 1). The API's job is to mint the grant,
rate-limit who may mint one, and report status.

The grant is a POST policy, not a PUT URL, so the size cap is a signed
condition enforced by object storage. Since the bytes never reach the gateway,
a PUT grant would make DF_MAX_UPLOAD_BYTES unenforceable.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from df import storage as storage_mod
from df.config import settings
from df.db import Db
from df.inference.base import (
    VALIDATION_PLACEHOLDER,
    VALIDATION_PRODUCTION,
    VALIDATION_RESEARCH,
)
from df.inference.calibration import SCHEME_LAUNCH_SNAPSHOT, SCHEME_UNFITTED
from df.rollup import face_evidence
from df.jobstatus import JobStatus, assert_persistence_enabled
from df.queue import TOPIC_PREPROCESS, Queue, build_queue
from df.ratelimit import RateLimiter

log = logging.getLogger("df.gateway")

app = FastAPI(title="Deepfake Detection API", version="0.1.0")

_db = Db()
_storage: storage_mod.Storage | None = None
_queue: Queue | None = None
_status: JobStatus | None = None
_limiter: RateLimiter | None = None

MEDIA_TYPES = {"video", "image", "audio"}


@app.on_event("startup")
def _startup() -> None:
    global _storage, _queue, _status, _limiter
    _storage = storage_mod.build_storage()
    _queue = build_queue()
    _status = JobStatus()
    _limiter = RateLimiter()
    # Fail fast rather than discover on the first restart that every in-flight
    # job vanished.
    assert_persistence_enabled(_status.r)


def identity_of(request: Request) -> str:
    """Rate-limit key. API key when present, client IP otherwise.

    NOTE: behind a proxy this must read the trusted forwarded-for header instead
    of the socket peer, or every request buckets to the proxy. Wire that up with
    the ingress config -- do not trust X-Forwarded-For unconditionally.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key[:32]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def enforce_rate_limit(request: Request, cost: float = 1.0) -> str:
    ident = identity_of(request)
    decision = _limiter.check(ident, cost=cost)
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(decision.retry_after_seconds) + 1)},
        )
    return ident


class CreateJobRequest(BaseModel):
    media_type: str = Field(description="video | image | audio")
    content_type: str = Field(default="application/octet-stream")
    size_bytes: int | None = Field(default=None)


class CreateJobResponse(BaseModel):
    job_id: str
    upload_url: str
    upload_method: str = "POST"
    # Policy fields the client must send as form parts, before the file part.
    # They carry the signature and the content-length-range condition, so the
    # upload is size-capped by object storage rather than on the client's word.
    upload_fields: dict[str, str] = Field(default_factory=dict)
    expires_in_seconds: int
    max_bytes: int
    notify_url: str


@app.post("/v1/jobs", response_model=CreateJobResponse, status_code=201)
def create_job(body: CreateJobRequest, request: Request) -> CreateJobResponse:
    if body.media_type not in MEDIA_TYPES:
        raise HTTPException(400, f"media_type must be one of {sorted(MEDIA_TYPES)}")
    if body.size_bytes is not None and body.size_bytes > settings.max_upload_bytes:
        raise HTTPException(413, f"upload exceeds {settings.max_upload_bytes} bytes")

    ident = enforce_rate_limit(request)

    job_id = _db.create_job(
        media_type=body.media_type,
        raw_object_key="",              # filled below, needs the id
        derived_prefix="",
        submitted_by=ident,
    )
    raw_key = storage_mod.raw_key(job_id)
    derived = storage_mod.derived_prefix(job_id)
    with _db.conn() as c, c.transaction():
        c.execute(
            "UPDATE jobs SET raw_object_key = %s, derived_prefix = %s WHERE id = %s",
            (raw_key, derived, job_id),
        )

    grant = _storage.presign_upload(raw_key, body.content_type, settings.max_upload_bytes)
    _status.publish(job_id, "awaiting_upload")
    _db.record_event(job_id, "job.created", {"media_type": body.media_type, "by": ident})

    return CreateJobResponse(
        job_id=job_id,
        upload_url=grant.url,
        upload_method=grant.method,
        upload_fields=grant.fields,
        expires_in_seconds=settings.presign_ttl_seconds,
        max_bytes=settings.max_upload_bytes,
        notify_url=f"/v1/jobs/{job_id}/uploaded",
    )


@app.post("/v1/jobs/{job_id}/uploaded", status_code=202)
def mark_uploaded(job_id: str, request: Request) -> dict:
    """Client calls this after the presigned upload POST succeeds; enqueues work.

    POST, not PUT -- see the module docstring and storage.presign_upload. The
    grant is a POST policy precisely so the size cap is a signed condition
    storage enforces; saying "PUT" here invites someone to build one.
    """
    enforce_rate_limit(request)

    job = _db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "awaiting_upload":
        # Idempotent: a retried notify must not enqueue the job twice.
        return {"job_id": job_id, "status": job["status"], "enqueued": False}

    if not _storage.exists(job["raw_object_key"]):
        raise HTTPException(409, "upload not found in storage")

    _db.set_status(job_id, "queued")
    _status.publish(job_id, "queued")
    _queue.push(TOPIC_PREPROCESS, {"job_id": job_id, "media_type": job["media_type"]})
    _db.record_event(job_id, "job.enqueued", {})
    return {"job_id": job_id, "status": "queued", "enqueued": True}


def _decode_advisories(preprocess: dict) -> list[str]:
    """Say when nothing could be READ, rather than reporting no faces.

    `undetermined` is a legitimate verdict for a photo with nobody in it. It is
    the wrong verdict for a photo the service could not open, and the two were
    indistinguishable: an undecodable upload produced zero items and routed to
    `undetermined` exactly as an empty room would.

    That mattered most in the likeliest case. A phone camera roll is HEIC by
    default, OpenCV has no HEIF codec at all (`measured: yes` 2026-08-31 -- its
    build lists JPEG, PNG and WEBP only), so someone uploading a picture of their
    own face was told no face was found in it. A confidently wrong answer, and
    nothing about it looked broken.
    """
    if preprocess.get("decodable") is not False:
        return []
    fmt = preprocess.get("media_format") or "unknown"
    return [
        f"MEDIA NOT DECODED: the upload was detected as {fmt!r}, which this "
        "service could not open, so no faces were examined. This result means "
        "'not analysed', NOT 'no face present'. Re-upload as JPEG or PNG."
    ]


def _calibration_advisories(job: dict) -> list[str]:
    """Caveat about the SCALE of the number, not the trustworthiness of the
    weights. FAILS CLOSED, like its neighbour.

    These are separate questions and both have to be answerable. A
    research-checkpoint warning says the weights were not validated here; it
    says nothing about whether the 0-100 figure is a calibrated probability. An
    unfitted temperature means it is not: the score is a raw sigmoid of a raw
    logit, so "82" is not "82% likely manipulated" and the score bands were
    drawn for a scale nothing has been mapped onto.

    Silent for the stub, whose PLACEHOLDER caveat already says the number is
    meaningless -- adding a second warning about the calibration of a
    meaningless number is noise that trains readers to skip advisories.
    """
    if job.get("model_validation") == VALIDATION_PLACEHOLDER:
        return []

    calibration = job.get("calibration")

    if calibration == SCHEME_LAUNCH_SNAPSHOT:
        return [
            "LAUNCH-SNAPSHOT CALIBRATION: the temperature was fitted once, on a "
            "held-out set captured at launch. It is not tracked for drift and is "
            "not re-fitted per retrain."
        ]

    if calibration == SCHEME_UNFITTED:
        return [
            "UNCALIBRATED SCORE: no temperature has been fitted for this model, "
            "so this number is a raw model output, not a calibrated probability. "
            "It is not a percentage likelihood, and the score bands were not "
            "drawn against this scale."
        ]

    # NULL, or a scheme this function does not know about. Same rule as the
    # validation advisory: unrecognised provenance warns rather than stays quiet.
    return [
        "UNRECORDED CALIBRATION: nothing recorded how this score was calibrated, "
        "so it cannot be compared with other results or read as a probability."
    ]


def _validation_advisories(job: dict) -> list[str]:
    """Caveats about how far this score can be trusted. FAILS CLOSED.

    The previous version matched the substring "stub" against model_version_id.
    That failed OPEN: loading any real checkpoint changed the id to something
    like face-efficientnet_b4-<hash>, the substring vanished, and with it every
    caveat -- silently, at the exact moment scores start looking plausible
    enough to be believed. Whether a reader is warned must not depend on how a
    model happens to be named.

    So this reads the level recorded on the job row at decision time, and any
    value it does not recognise -- including NULL, including a level added later
    by someone who did not update this function -- produces the strongest
    caveat rather than none. Unknown provenance is a reason to warn.
    """
    level = job.get("model_validation")

    if level == VALIDATION_PRODUCTION:
        return []

    if level == VALIDATION_PLACEHOLDER:
        return [
            "PLACEHOLDER MODEL: this score was produced by a stub scorer, not a "
            "trained detector. It carries no detection meaning."
        ]

    if level == VALIDATION_RESEARCH:
        return [
            "RESEARCH CHECKPOINT: this score came from a public pre-trained model "
            "evaluated on a public slice, not from weights validated for this "
            "system. It has not been measured against this pipeline's score "
            "bands, and its training data may not resemble your input. Treat it "
            "as indicative, not as a finding."
        ]

    # NULL, or anything unrecognised.
    return [
        "UNVERIFIED MODEL: no validation level was recorded for the weights that "
        "produced this score, so nothing here establishes what they were or how "
        "far they can be trusted. Treat this result as unverified."
    ]


def _coverage(job: dict) -> float | None:
    """item_count / items_total, or None when either was never measured.

    None is not 0.0 and must not be coerced to it: a pre-006 row never recorded
    the denominator, and a caller that reads a missing coverage as "no coverage"
    would treat every historical result as worthless rather than as unmeasured.
    """
    total = job.get("items_total")
    used = job.get("item_count")
    if not total or used is None:
        return None
    return round(used / total, 4)


def _public_job(job: dict, _preprocess: dict | None = None) -> dict:
    doc = {
        "job_id": str(job["id"]),
        "media_type": job["media_type"],
        "status": job["status"],
        "result_class": job["result_class"],
        "band": job["band"],
        "aggregate_score": job["aggregate_score"],
        "face_count": job["face_count"],
        "item_count": job["item_count"],
        # What was extracted before confidence drops and trimming, and what
        # fraction of it survived to produce the score. Reported on EVERY
        # verdict, at every k: a one-frame verdict and a fifty-frame verdict are
        # otherwise indistinguishable in this response, and the answer to that
        # is to mark the difference rather than to withhold the verdict.
        # NULL means never measured (a row written before migration 006).
        "items_total": job.get("items_total"),
        "coverage": _coverage(job),
        # How the raw logit became this score. NULL means not recorded
        # (a pre-007 row), which is not the same as "no calibration".
        "calibration": job.get("calibration"),
        "model_version_id": job["model_version_id"],
        "aggregation_method": job["aggregation_method"],
        "aggregation_params": job["aggregation_params"],
        "content_hash": job["content_hash"],
        # NULL means never measured (a row written before migration 004), which
        # is not the same claim as 0. Passed through rather than coerced.
        "items_unattributed": job.get("items_unattributed"),
        # NULL means never recorded, which callers must read as untrusted.
        "model_validation": job.get("model_validation"),
        "created_at": job["created_at"].isoformat() if job["created_at"] else None,
        "completed_at": job["completed_at"].isoformat() if job["completed_at"] else None,
        "media_deleted": job["raw_deleted_at"] is not None,
        "extended_retention_until": (
            job["extended_retention_until"].isoformat()
            if job["extended_retention_until"] else None
        ),
    }
    # A failed job has to say why. The WebSocket already publishes the reason
    # on the failure event, but this polling view omitted it -- so a client on
    # the reconnect fallback that Tier 1 mandates learned that its upload
    # failed and never what was wrong with it. Same string the socket sends,
    # so this exposes nothing the other channel did not already.
    if job["status"] in {"dead_letter", "failed"}:
        doc["failure_reason"] = job.get("error") or "processing failed"

    # Scores are manipulable by adversarial perturbation; there is no
    # adversarial-input pre-classifier (Tier 3, not built). Say so on every
    # result rather than burying it in docs.
    doc["advisories"] = [
        "Scores can be manipulated by adversarial perturbation of the input. "
        "No adversarial-input pre-classifier is in place.",
    ]
    doc["advisories"].extend(_validation_advisories(job))
    # model_version_id is the field a reader trusts to mean "these weights
    # produced this score". When part of the evidence had no recorded producer
    # that is true of most of the job, not all of it, and saying so is cheaper
    # than letting the id imply more than was observed.
    doc["advisories"].extend(_calibration_advisories(job))
    doc["advisories"].extend(_decode_advisories(_preprocess or {}))
    if job.get("items_unattributed"):
        doc["advisories"].append(
            f"PARTIAL PROVENANCE: {job['items_unattributed']} of this job's "
            f"scored items carry no recorded model version. The result is "
            f"attributed to {job['model_version_id']}, which produced the rest."
        )

    if job["extended_retention_until"]:
        doc["advisories"].append(
            "An extended retention window (fixed timer) is open on this result. "
            "The media that drove the score is retained until "
            f"{doc['extended_retention_until']}; the original upload was deleted "
            "on completion. The window expires automatically and is not a legal hold."
        )
    return doc


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> JSONResponse:
    """Polling fallback. Clients that lose the WebSocket read this on reconnect."""
    enforce_rate_limit(request, cost=0.1)

    live = _status.read(job_id)
    job = _db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    preprocess = _db.get_discarded_detections(job_id)
    doc = _public_job(job, preprocess)

    # Per-face detail behind a rolled-up label. Read from job_items rather than
    # denormalised onto the job row: those rows ARE the audit trail, they
    # outlive the media, and a second copy is a second thing that can drift.
    #
    # A crowd scene labelled `manipulated` because one background face scored
    # high is indistinguishable, in a scalar response, from a close-up of a
    # manipulated subject. This does not change the label -- it reports what the
    # label was reduced from, so the reduction is no longer the only artifact
    # and a consumer can apply its own bar. There is deliberately no per-face
    # threshold here; see df.rollup.face_evidence.
    if job["media_type"] in {"video", "image"} and job["status"] == "complete":
        doc["face_evidence"] = face_evidence(
            _db.get_items(job_id), discarded=preprocess
        )

    if live and live.get("status"):
        # Redis is ahead of Postgres between a transition and its commit.
        doc["live_status"] = live["status"]
    return JSONResponse(doc)


@app.websocket("/v1/jobs/{job_id}/ws")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    """Live status push.

    Sends the current status immediately on connect before streaming updates --
    a client reconnecting after a drop must not sit waiting for a transition
    that already happened.
    """
    await websocket.accept()

    import redis.asyncio as aredis

    client = aredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    await pubsub.subscribe(f"job:{job_id}:events")

    try:
        current = await client.get(f"job:{job_id}:status")
        if current:
            await websocket.send_text(current)

        while True:
            msg = await pubsub.get_message(timeout=30.0)
            if msg is None:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                continue
            await websocket.send_text(msg["data"])
            payload = json.loads(msg["data"])
            if payload.get("status") in {"complete", "failed", "dead_letter"}:
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await pubsub.unsubscribe()
        await pubsub.aclose()
        await client.aclose()
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "inference_backend": settings.inference_backend}
